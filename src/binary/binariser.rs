use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::iter::Sum;

use crate::utils::dist::*;

///////////////
// Binariser //
///////////////

const MAX_SAMPLES_PCA: usize = 100_000;

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
/// Supports SimHash (for Cosine similarity) and E2LSH (for Euclidean distance).
///
/// ### Fields
///
/// * `random_projections` - Random vectors from N(0,1), flattened (n_bits *
///   dim)
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

    /// Initialise binariser using PCA followed by ITQ rotation
    ///
    /// Uses Principal Component Analysis to find the directions of maximum variance,
    /// then applies Iterative Quantisation (ITQ) to rotate these components for
    /// optimal binary quantisation. This typically produces better quality codes
    /// than random projections.
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

    /// Prepare PCA projections with ITQ rotation for optimal binary quantisation
    ///
    /// Implements the algorithm from "Iterative Quantization: A Procrustean
    /// Approach to Learning Binary Codes for Large-scale Image Retrieval"
    /// (Gong et al., 2013).
    ///
    /// Steps:
    ///
    /// 1. Sample up to MAX_SAMPLES_PCA points if dataset is large
    /// 2. Centre the data by subtracting the mean
    /// 3. Compute PCA to find principal components
    /// 4. Apply ITQ to find optimal rotation for binary quantisation
    /// 5. If n_bits > dim, add orthogonalised random projections for remaining bits
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `seed` - Random seed for sampling and random projections
    /// * `itq_iterations` - Number of ITQ iterations (typically 10-50)
    ///   Prepare PCA projections with ITQ rotation for optimal binary quantisation
    ///
    /// Implements the algorithm from "Iterative Quantization: A Procrustean
    /// Approach to Learning Binary Codes for Large-scale Image Retrieval"
    /// (Gong et al., 2013).
    ///
    /// Steps:
    ///
    /// 1. Sample up to MAX_SAMPLES_PCA points if dataset is large
    /// 2. Centre the data by subtracting the mean
    /// 3. Compute PCA to find principal components
    /// 4. Apply ITQ to find optimal rotation for binary quantisation
    /// 5. If n_bits > dim, add normalised random projections for remaining bits
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `seed` - Random seed for sampling and random projections
    /// * `itq_iterations` - Number of ITQ iterations (typically 10-50)
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

        let mut r_mat = Mat::<T>::zeros(n_pca_bits, n_pca_bits);
        for i in 0..n_pca_bits {
            for j in 0..n_pca_bits {
                let val: f64 = rng.sample(StandardNormal);
                r_mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }
        // QR decomposition to get orthogonal matrix
        let qr = r_mat.as_ref().qr();
        r_mat = qr.compute_Q();

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
            let svd_itq = c_mat.as_ref().thin_svd().unwrap();

            r_mat = svd_itq.U() * svd_itq.V().transpose();
        }

        let final_projections_mat = v_pc * r_mat;
        let mut projections = Vec::with_capacity(self.n_bits * dim);

        for j in 0..n_pca_bits {
            for i in 0..dim {
                projections.push(final_projections_mat[(i, j)]);
            }
        }

        if self.n_bits > dim {
            for _ in n_pca_bits..self.n_bits {
                let mut proj = Vec::with_capacity(dim);
                for _ in 0..dim {
                    let val: f64 = rng.sample(StandardNormal);
                    proj.push(T::from_f64(val).unwrap());
                }

                // Just normalise
                let mut norm_sq = T::zero();
                for d in 0..dim {
                    norm_sq = norm_sq + proj[d] * proj[d];
                }
                let norm = norm_sq.sqrt();
                if norm > T::epsilon() {
                    for d in 0..dim {
                        proj[d] = proj[d] / norm;
                    }
                }

                projections.extend(proj);
            }
        }

        self.projections = projections;
    }
}

/////////////////////
// RaBitQQuantiser //
/////////////////////

/// Encoded query for RaBitQ distance estimation
///
/// ### Fields
///
/// * `quantised` - Int4 quantised values (one per dimension, stored as u8)
/// * `dist_to_centroid` - Distance from query to centroid
/// * `lower` - Lower bound used in quantisation
/// * `width` - Bucket width used in quantisation
/// * `sum_quantised` - Sum of all quantised values
pub struct RaBitQQuery<T> {
    pub quantised: Vec<u8>,
    pub dist_to_centroid: T,
    pub lower: T,
    pub width: T,
    pub sum_quantised: u32,
}

/// RaBitQ quantiser
///
/// Transforms vectors relative to a centroid and encodes them as binary codes
/// with correction factors for accurate distance estimation. This version is
/// based on findings from ElasticSearch labs in terms of number of centroids,
/// rotations, etc.
///
/// ### Fields
///
/// * `centroid` - Centroid of the training data
/// * `dim` - Input dimensionality
/// * `metric` - The distance metric used
pub struct RaBitQQuantiser<T> {
    pub centroid: Vec<T>,
    pub rotation: Vec<T>,
    pub dim: usize,
    pub metric: Dist,
}

impl<T> RaBitQQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Create a new RaBitQ quantiser from training data
    ///
    /// Computes the centroid from all vectors. For cosine distance, vectors are
    /// normalised before computing the centroid.
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `metric` - Dist metric (Euclidean or Cosine)
    ///
    /// ### Returns
    ///
    /// Initialised quantiser
    pub fn new(data: MatRef<T>, metric: &Dist, seed: usize) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        let mut centroid = vec![T::zero(); dim];

        for i in 0..n {
            let row = data.row(i);
            let (scale, vec): (T, Vec<T>) = match metric {
                Dist::Cosine => {
                    let norm = compute_norm_row(row);
                    if norm > T::epsilon() {
                        (norm, row.iter().cloned().collect())
                    } else {
                        (T::one(), row.iter().cloned().collect())
                    }
                }
                Dist::Euclidean => (T::one(), row.iter().cloned().collect()),
            };

            for d in 0..dim {
                centroid[d] = centroid[d] + vec[d] / scale;
            }
        }

        let n_t = T::from_usize(n).unwrap();
        for d in 0..dim {
            centroid[d] = centroid[d] / n_t;
        }

        let rotation = Self::generate_random_orthogonal(dim, seed as u64);

        Self {
            centroid,
            rotation,
            dim,
            metric: *metric,
        }
    }

    /// Encode a vector for indexing
    ///
    /// Returns binary code and correction factors.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector
    ///
    /// ### Returns
    ///
    /// Tuple of (binary_code, dist_to_centroid, dot_correction)
    #[inline]
    pub fn encode_vec(&self, vec: &[T]) -> (Vec<u8>, T, T) {
        assert_eq!(vec.len(), self.dim);

        let vec_norm: Vec<T> = match self.metric {
            Dist::Cosine => {
                let norm = compute_norm(vec);
                if norm > T::epsilon() {
                    vec.iter().map(|&x| x / norm).collect()
                } else {
                    vec.to_vec()
                }
            }
            Dist::Euclidean => vec.to_vec(),
        };

        // Compute residual relative to centroid
        let mut res = vec![T::zero(); self.dim];
        for d in 0..self.dim {
            res[d] = vec_norm[d] - self.centroid[d];
        }

        let dist_to_centroid = compute_norm(&res);

        // Normalise residual to unit vector v_c
        let mut v_c = vec![T::zero(); self.dim];
        if dist_to_centroid > T::epsilon() {
            for d in 0..self.dim {
                v_c[d] = res[d] / dist_to_centroid;
            }
        }

        // Apply rotation
        let v_c_rotated = self.rotate(&v_c);

        // Binary encode: Store the sign of the rotated components
        let n_bytes = self.dim.div_ceil(8);
        let mut binary = vec![0u8; n_bytes];
        for d in 0..self.dim {
            if v_c_rotated[d] >= T::zero() {
                binary[d / 8] |= 1u8 << (d % 8);
            }
        }

        // Dot correction: ||v_c_rotated||_1
        // This represents the sum of absolute values of the unit residual components.
        // In RaBitQ, this is used to scale the binary dot product back to an inner product estimate.
        let mut dot_correction = T::zero();
        for d in 0..self.dim {
            dot_correction = dot_correction + v_c_rotated[d].abs();
        }

        (binary, dist_to_centroid, dot_correction)
    }

    /// Encode a query vector
    ///
    /// Transforms and scalar quantises to int4.
    ///
    /// ### Params
    ///
    /// * `vec` - Query vector
    ///
    /// ### Returns
    ///
    /// Encoded query with all metadata for distance estimation
    pub fn encode_query(&self, vec: &[T]) -> RaBitQQuery<T> {
        assert_eq!(vec.len(), self.dim);

        let vec_norm: Vec<T> = match self.metric {
            Dist::Cosine => {
                let norm = compute_norm(vec);
                if norm > T::epsilon() {
                    vec.iter().map(|&x| x / norm).collect()
                } else {
                    vec.to_vec()
                }
            }
            Dist::Euclidean => vec.to_vec(),
        };

        // compute residual: v - c
        let mut res = vec![T::zero(); self.dim];
        for d in 0..self.dim {
            res[d] = vec_norm[d] - self.centroid[d];
        }

        // compute dist to centroid: ||v - c||
        let dist_to_centroid = compute_norm(&res);

        // normalise residual: v_c = (v - c) / ||v - c||
        let mut q_c = vec![T::zero(); self.dim];
        if dist_to_centroid > T::epsilon() {
            for d in 0..self.dim {
                q_c[d] = res[d] / dist_to_centroid;
            }
        }

        // Apply same rotation
        let q_c_rotated = self.rotate(&q_c);

        // Scalar quantise the rotated vector
        let mut lower = q_c_rotated[0];
        let mut upper = q_c_rotated[0];
        for d in 1..self.dim {
            if q_c_rotated[d] < lower {
                lower = q_c_rotated[d];
            }
            if q_c_rotated[d] > upper {
                upper = q_c_rotated[d];
            }
        }

        let range = upper - lower;
        let width = if range > T::epsilon() {
            range / T::from_f32(15.0).unwrap()
        } else {
            T::one()
        };

        let mut quantised = vec![0u8; self.dim];
        let mut sum_quantised: u32 = 0;

        for d in 0..self.dim {
            let val = ((q_c_rotated[d] - lower) / width)
                .round()
                .to_u8()
                .unwrap_or(0)
                .min(15);
            quantised[d] = val;
            sum_quantised += val as u32;
        }

        RaBitQQuery {
            quantised,
            dist_to_centroid,
            lower,
            width,
            sum_quantised,
        }
    }

    /// Memory usage in bytes
    ///
    /// ### Returns
    ///
    /// The memory fingerprint in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.centroid.capacity() * std::mem::size_of::<T>()
    }

    /// Generate a random orthogonal matrix
    fn generate_random_orthogonal(dim: usize, seed: u64) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed);

        // Create random matrix
        let mut mat = Mat::<T>::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let val: f64 = rng.sample(StandardNormal);
                mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }

        // QR decomposition gives orthogonal Q
        let qr = mat.as_ref().qr();
        let q = qr.compute_Q();

        // Flatten row-major for fast dot products
        let mut rotation = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                rotation.push(q[(i, j)]);
            }
        }
        rotation
    }

    /// Apply rotation to a vector
    #[inline]
    fn rotate(&self, vec: &[T]) -> Vec<T> {
        let mut rotated = vec![T::zero(); self.dim];
        for i in 0..self.dim {
            let base = i * self.dim;
            let mut sum = T::zero();
            for j in 0..self.dim {
                sum = sum + self.rotation[base + j] * vec[j];
            }
            rotated[i] = sum;
        }
        rotated
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
